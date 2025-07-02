"""
Hardware Compatibility Testing

Tests deployment packages on different hardware platforms to ensure
compatibility and performance.
"""

import json
import logging
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class HardwareProfile:
    """Hardware profile information"""

    name: str
    architecture: str
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    storage_gb: float
    gpu_available: bool
    gpu_model: Optional[str] = None
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""


@dataclass
class CompatibilityTest:
    """Hardware compatibility test"""

    name: str
    description: str
    test_function: callable
    timeout: int = 60
    required_features: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result"""

    test_name: str
    status: TestStatus
    duration: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HardwareDetector:
    """Detects current hardware configuration"""

    def detect_hardware(self) -> HardwareProfile:
        """Detect current hardware profile"""
        # Basic system info
        arch = platform.machine()
        cpu_count = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / (1024**3)

        # Storage info
        storage_gb = psutil.disk_usage("/").total / (1024**3)

        # CPU model
        cpu_model = self._get_cpu_model()

        # GPU detection
        gpu_available, gpu_model = self._detect_gpu()

        # OS info
        os_name = platform.system()
        os_version = platform.release()

        # Python version
        python_version = sys.version.split()[0]

        # Determine profile name
        profile_name = self._determine_profile_name(arch, ram_gb, gpu_available)

        return HardwareProfile(
            name=profile_name,
            architecture=arch,
            cpu_model=cpu_model,
            cpu_cores=cpu_count,
            ram_gb=round(ram_gb, 2),
            storage_gb=round(storage_gb, 2),
            gpu_available=gpu_available,
            gpu_model=gpu_model,
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
        )

    def _get_cpu_model(self) -> str:
        """Get CPU model name"""
        try:
            if platform.system() == "Darwin":
                # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()
            elif platform.system() == "Linux":
                # Linux
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Windows":
                # Windows
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], capture_output=True, text=True
                )
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    return lines[1].strip()
        except Exception:
            pass

        return "Unknown CPU"

    def _detect_gpu(self) -> tuple[bool, Optional[str]]:
        """Detect GPU availability and model"""
        try:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                return True, gpu_name
        except FileNotFoundError:
            pass

        try:
            # Try for Apple Silicon GPU
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                )
                if "Apple" in result.stdout and "GPU" in result.stdout:
                    return True, "Apple Silicon GPU"
        except Exception:
            pass

        return False, None

    def _determine_profile_name(
            self,
            arch: str,
            ram_gb: float,
            gpu_available: bool) -> str:
        """Determine profile name based on hardware"""
        if arch == "arm64" or arch == "aarch64":
            if ram_gb <= 2:
                return "raspberry_pi"
            elif ram_gb <= 8 and gpu_available:
                return "jetson_nano"
            elif platform.system() == "Darwin":
                return "mac_mini_m2"
            else:
                return "generic_arm64"
        elif arch == "x86_64":
            if gpu_available:
                return "x86_64_gpu"
            else:
                return "x86_64_cpu"
        else:
            return "unknown"


class CompatibilityTester:
    """
    Runs compatibility tests on hardware platforms.
    """

    def __init__(self) -> None:
        """Initialize compatibility tester"""
        self.detector = HardwareDetector()
        self.tests = self._create_test_suite()

    def _create_test_suite(self) -> List[CompatibilityTest]:
        """Create test suite"""
        return [
            CompatibilityTest(
                name="cpu_performance",
                description="Test CPU performance",
                test_function=self._test_cpu_performance,
                timeout=30,
            ),
            CompatibilityTest(
                name="memory_allocation",
                description="Test memory allocation",
                test_function=self._test_memory_allocation,
                timeout=20,
            ),
            CompatibilityTest(
                name="disk_io",
                description="Test disk I/O performance",
                test_function=self._test_disk_io,
                timeout=30,
            ),
            CompatibilityTest(
                name="python_packages",
                description="Test Python package compatibility",
                test_function=self._test_python_packages,
                timeout=60,
            ),
            CompatibilityTest(
                name="network_connectivity",
                description="Test network connectivity",
                test_function=self._test_network,
                timeout=20,
            ),
            CompatibilityTest(
                name="gpu_availability",
                description="Test GPU availability",
                test_function=self._test_gpu,
                timeout=10,
                required_features=["gpu"],
            ),
            CompatibilityTest(
                name="agent_startup",
                description="Test agent startup",
                test_function=self._test_agent_startup,
                timeout=120,
            ),
            CompatibilityTest(
                name="resource_limits",
                description="Test resource limits",
                test_function=self._test_resource_limits,
                timeout=30,
            ),
        ]

    def run_tests(
        self, package_dir: Path, hardware_profile: Optional[HardwareProfile] = None
    ) -> Dict[str, Any]:
        """
        Run compatibility tests.

        Args:
            package_dir: Package directory to test
            hardware_profile: Hardware profile (auto-detect if None)

        Returns:
            Test results
        """
        if not hardware_profile:
            hardware_profile = self.detector.detect_hardware()

        logger.info(f"Running compatibility tests on {hardware_profile.name}")

        results = []
        start_time = time.time()

        for test in self.tests:
            # Check if test should run
            if test.required_features:
                skip = False
                for feature in test.required_features:
                    if feature == "gpu" and not hardware_profile.gpu_available:
                        skip = True
                        break

                if skip:
                    results.append(
                        TestResult(
                            test_name=test.name,
                            status=TestStatus.SKIPPED,
                            duration=0,
                            message=(
                                f"Skipped: requires {
                                    ', '.join(
                                        test.required_features)}",
                            ),
                        ))
                    continue

            # Run test
            result = self._run_test(test, package_dir, hardware_profile)
            results.append(result)

        total_duration = time.time() - start_time

        # Generate summary
        summary = {
            "hardware_profile": hardware_profile.__dict__,
            "test_results": [r.__dict__ for r in results],
            "summary": {
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.status == TestStatus.PASSED),
                "failed": sum(1 for r in results if r.status == TestStatus.FAILED),
                "skipped": sum(1 for r in results if r.status == TestStatus.SKIPPED),
                "duration": total_duration,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return summary

    def _run_test(
        self,
        test: CompatibilityTest,
        package_dir: Path,
        hardware_profile: HardwareProfile,
    ) -> TestResult:
        """Run a single test with timeout"""
        logger.info(f"Running test: {test.name}")

        result_queue = queue.Queue()

        def test_wrapper() -> None:
            try:
                start_time = time.time()
                success, message, details = test.test_function(
                    package_dir, hardware_profile)
                duration = time.time() - start_time

                status = TestStatus.PASSED if success else TestStatus.FAILED
                result = TestResult(
                    test_name=test.name,
                    status=status,
                    duration=duration,
                    message=message,
                    details=details,
                )
                result_queue.put(result)

            except Exception as e:
                result = TestResult(
                    test_name=test.name,
                    status=TestStatus.FAILED,
                    duration=0,
                    message=f"Test error: {str(e)}",
                    error=str(e),
                )
                result_queue.put(result)

        # Run test in thread with timeout
        thread = threading.Thread(target=test_wrapper)
        thread.start()
        thread.join(timeout=test.timeout)

        if thread.is_alive():
            # Test timed out
            return TestResult(
                test_name=test.name,
                status=TestStatus.TIMEOUT,
                duration=test.timeout,
                message=f"Test timed out after {test.timeout}s",
            )

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return TestResult(
                test_name=test.name,
                status=TestStatus.FAILED,
                duration=0,
                message="Test completed but no result returned",
            )

    def _test_cpu_performance(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test CPU performance"""
        import numpy as np

        # Simple CPU benchmark
        size = 1000
        iterations = 100

        start_time = time.time()

        for _ in range(iterations):
            # Matrix multiplication
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            c = np.dot(a, b)

            # Some computation to prevent optimization
            _ = np.sum(c)

        duration = time.time() - start_time
        ops_per_second = (iterations * size * size * size * 2) / \
            duration / 1e9  # GFLOPS

        # Determine if performance is acceptable
        min_gflops = {
            "raspberry_pi": 0.5,
            "jetson_nano": 2.0,
            "mac_mini_m2": 10.0,
            "x86_64_cpu": 5.0,
            "x86_64_gpu": 5.0,
        }.get(hardware_profile.name, 1.0)

        passed = ops_per_second >= min_gflops

        return (
            passed,
            f"CPU performance: {ops_per_second:.2f} GFLOPS",
            {
                "gflops": ops_per_second,
                "duration": duration,
                "min_required": min_gflops,
            },
        )

    def _test_memory_allocation(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test memory allocation"""
        # Test allocating different amounts of memory
        test_sizes_mb = [10, 50, 100, 500]
        max_allocatable = 0

        for size_mb in test_sizes_mb:
            if size_mb > hardware_profile.ram_gb * 1024 * 0.5:  # Don't exceed 50% of RAM
                break

            try:
                # Allocate memory
                size_bytes = size_mb * 1024 * 1024
                data = bytearray(size_bytes)

                # Write pattern to ensure allocation
                for i in range(0, len(data), 1024):
                    data[i] = i % 256

                max_allocatable = size_mb
                del data  # Free memory

            except MemoryError:
                break

        # Check if we can allocate minimum required
        min_required_mb = {
            "raspberry_pi": 50,
            "jetson_nano": 200,
            "mac_mini_m2": 500,
            "x86_64_cpu": 200,
            "x86_64_gpu": 500,
        }.get(hardware_profile.name, 100)

        passed = max_allocatable >= min_required_mb

        return (
            passed,
            f"Maximum allocatable memory: {max_allocatable}MB",
            {
                "max_allocatable_mb": max_allocatable,
                "min_required_mb": min_required_mb,
                "total_ram_gb": hardware_profile.ram_gb,
            },
        )

    def _test_disk_io(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test disk I/O performance"""
        test_file = package_dir / "io_test.tmp"
        file_size_mb = 100

        try:
            # Write test
            data = bytearray(1024 * 1024)  # 1MB chunks
            start_time = time.time()

            with open(test_file, "wb") as f:
                for _ in range(file_size_mb):
                    f.write(data)

            write_duration = time.time() - start_time
            write_speed = file_size_mb / write_duration

            # Read test
            start_time = time.time()

            with open(test_file, "rb") as f:
                while f.read(1024 * 1024):
                    pass

            read_duration = time.time() - start_time
            read_speed = file_size_mb / read_duration

            # Cleanup
            test_file.unlink()

            # Check minimum speeds
            min_write_speed = 10  # MB/s
            min_read_speed = 20  # MB/s

            passed = write_speed >= min_write_speed and read_speed >= min_read_speed

            return (
                passed,
                f"Disk I/O: Write {write_speed:.1f}MB/s, Read {read_speed:.1f}MB/s",
                {
                    "write_speed_mbps": write_speed,
                    "read_speed_mbps": read_speed,
                    "file_size_mb": file_size_mb,
                },
            )

        except Exception as e:
            return (False, f"Disk I/O test failed: {str(e)}", {"error": str(e)})
        finally:
            if test_file.exists():
                test_file.unlink()

    def _test_python_packages(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test Python package compatibility"""
        requirements_file = package_dir / "requirements.txt"

        if not requirements_file.exists():
            return (True, "No requirements.txt found", {"skipped": True})

        try:
            # Check if packages can be imported
            with open(requirements_file) as f:
                requirements = f.read().strip().split("\n")

            failed_imports = []
            successful_imports = []

            for req in requirements:
                if req and not req.startswith("#"):
                    # Extract package name
                    package_name = req.split("==")[0].split(">=")[0].split("[")[0]

                    # Try to import
                    try:
                        __import__(package_name.replace("-", "_"))
                        successful_imports.append(package_name)
                    except ImportError:
                        failed_imports.append(package_name)

            if failed_imports:
                return (
                    False,
                    f"Failed to import: {', '.join(failed_imports[:5])}",
                    {
                        "failed_imports": failed_imports,
                        "successful_imports": successful_imports,
                    },
                )
            else:
                return (
                    True,
                    f"All {len(successful_imports)} packages available",
                    {"successful_imports": successful_imports},
                )

        except Exception as e:
            return (False, f"Package test failed: {str(e)}", {"error": str(e)})

    def _test_network(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test network connectivity"""
        try:
            import socket

            # Test DNS resolution
            start_time = time.time()
            socket.gethostbyname("google.com")
            dns_time = time.time() - start_time

            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)

            start_time = time.time()
            result = sock.connect_ex(("google.com", 80))
            connect_time = time.time() - start_time
            sock.close()

            if result == 0:
                return (True, f"Network OK (DNS: {dns_time *
                                                  1000:.0f}ms, Connect: {connect_time *
                                                                         1000:.0f}ms)", {"dns_time_ms": dns_time *
                                                                                         1000, "connect_time_ms": connect_time *
                                                                                         1000, }, )
            else:
                return (False, "Network connection failed", {"error_code": result})

        except Exception as e:
            return (False, f"Network test failed: {str(e)}", {"error": str(e)})

    def _test_gpu(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test GPU availability and functionality"""
        if not hardware_profile.gpu_available:
            return (False, "No GPU detected", {"gpu_available": False})

        try:
            # Try to use GPU with common frameworks
            gpu_tests = []

            # Test PyTorch
            try:
                import torch

                cuda_available = torch.cuda.is_available()
                device_count = torch.cuda.device_count() if cuda_available else 0
                gpu_tests.append(
                    {
                        "framework": "pytorch",
                        "available": cuda_available,
                        "device_count": device_count,
                    }
                )
            except ImportError:
                pass

            # Test TensorFlow
            try:
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                gpu_tests.append(
                    {
                        "framework": "tensorflow",
                        "available": len(gpu_devices) > 0,
                        "device_count": len(gpu_devices),
                    }
                )
            except ImportError:
                pass

            # Test JAX
            try:
                import jax

                devices = jax.devices()
                gpu_count = sum(1 for d in devices if d.device_kind != "cpu")
                gpu_tests.append(
                    {
                        "framework": "jax",
                        "available": gpu_count > 0,
                        "device_count": gpu_count,
                    }
                )
            except ImportError:
                pass

            if gpu_tests:
                any_available = any(t["available"] for t in gpu_tests)
                return (
                    any_available,
                    f"GPU support: {len([t for t in gpu_tests if t['available']])} of {len(gpu_tests)} frameworks",
                    {
                        "gpu_model": hardware_profile.gpu_model,
                        "framework_tests": gpu_tests,
                    },
                )
            else:
                return (
                    True,
                    "GPU detected but no ML frameworks installed",
                    {"gpu_model": hardware_profile.gpu_model},
                )

        except Exception as e:
            return (False, f"GPU test failed: {str(e)}", {"error": str(e)})

    def _test_agent_startup(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test agent startup"""
        run_script = package_dir / "run.sh"

        if not run_script.exists():
            return (False, "No run.sh script found", {"script_exists": False})

        try:
            # Start agent process
            env = os.environ.copy()
            env["FREEAGENTICS_TEST_MODE"] = "1"  # Run in test mode

            process = subprocess.Popen(
                [str(run_script)],
                cwd=package_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for startup (max 30 seconds)
            startup_timeout = 30
            start_time = time.time()
            started = False

            while time.time() - start_time < startup_timeout:
                # Check if process is still running
                if process.poll() is not None:
                    # Process exited
                    stdout, stderr = process.communicate()
                    return (
                        False,
                        f"Agent exited with code {process.returncode}",
                        {
                            "returncode": process.returncode,
                            "stdout": stdout[-500:],
                            "stderr": stderr[-500:],
                        },
                    )

                # Check for startup indicator (could check log file or port)
                if (package_dir / "agent.pid").exists():
                    started = True
                    break

                time.sleep(1)

            if started:
                # Successfully started - terminate
                process.terminate()
                process.wait(timeout=5)

                return (
                    True,
                    f"Agent started successfully in {time.time() - start_time:.1f}s",
                    {"startup_time": time.time() - start_time},
                )
            else:
                # Timeout
                process.terminate()
                process.wait(timeout=5)

                return (False, "Agent startup timeout", {"timeout": startup_timeout})

        except Exception as e:
            return (False, f"Agent startup test failed: {str(e)}", {"error": str(e)})

    def _test_resource_limits(
        self, package_dir: Path, hardware_profile: HardwareProfile
    ) -> tuple[bool, str, dict]:
        """Test resource limits and constraints"""
        # Check if system has resource limits
        try:
            import resource

            limits = {
                "max_cpu_time": resource.getrlimit(resource.RLIMIT_CPU),
                "max_memory": resource.getrlimit(resource.RLIMIT_AS),
                "max_processes": resource.getrlimit(resource.RLIMIT_NPROC),
                "max_open_files": resource.getrlimit(resource.RLIMIT_NOFILE),
            }

            # Check if limits are reasonable
            issues = []

            if limits["max_memory"][0] != - \
                    1 and limits["max_memory"][0] < 1024 * 1024 * 1024:
                issues.append("Memory limit too low")

            if limits["max_open_files"][0] < 1024:
                issues.append("File descriptor limit too low")

            if limits["max_processes"][0] != -1 and limits["max_processes"][0] < 100:
                issues.append("Process limit too low")

            if issues:
                return (
                    False,
                    f"Resource limit issues: {', '.join(issues)}",
                    {"limits": limits, "issues": issues},
                )
            else:
                return (True, "Resource limits OK", {"limits": limits})

        except ImportError:
            # Windows doesn't have resource module
            return (
                True,
                "Resource limits not applicable on this platform",
                {"platform": platform.system()},
            )
        except Exception as e:
            return (False, f"Resource limit test failed: {str(e)}", {"error": str(e)})


def test_hardware_compatibility(package_path: str) -> bool:
    """
    Test hardware compatibility for a package.

    Args:
        package_path: Path to package directory

    Returns:
        True if all tests passed
    """
    tester = CompatibilityTester()
    package_dir = Path(package_path)

    if not package_dir.exists():
        logger.error(f"Package directory not found: {package_path}")
        return False

    # Run tests
    results = tester.run_tests(package_dir)

    # Print results
    print("\n=== Hardware Compatibility Test Results ===")
    print(f"Hardware: {results['hardware_profile']['name']}")
    print(f"Architecture: {results['hardware_profile']['architecture']}")
    print(f"CPU: {results['hardware_profile']['cpu_model']}")
    print(f"RAM: {results['hardware_profile']['ram_gb']}GB")
    print(f"GPU: {results['hardware_profile']['gpu_model'] or 'None'}")
    print()

    # Print test results
    for test_result in results["test_results"]:
        icon = {"passed": "✓", "failed": "✗", "skipped": "○", "timeout": "⏱"}.get(
            test_result["status"], "?"
        )

        print(f"{icon} {test_result['test_name']}: {test_result['message']}")

    # Summary
    summary = results["summary"]
    print(f"\nSummary: {summary['passed']}/{summary['total_tests']} passed")
    print(f"Duration: {summary['duration']:.1f}s")

    # Save detailed report
    report_file = package_dir / "compatibility_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    return summary["failed"] == 0
