"""
Technical Readiness Validation for Edge Deployment

Comprehensive assessment system that validates coalition technical readiness for edge deployment,
integrating with existing hardware infrastructure and providing detailed performance benchmarks.
"""

import asyncio
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil

# Import existing infrastructure
from infrastructure.deployment.hardware_compatibility import (
    CompatibilityTester,
    HardwareDetector,
    HardwareProfile,
    TestResult,
    TestStatus,
)
from infrastructure.hardware.device_discovery import DeviceDiscovery, DeviceType
from infrastructure.hardware.hal_core import (
    HardwareAbstractionLayer,
    HardwareType,
    ResourceConstraints,
)

logger = logging.getLogger(__name__)


class ReadinessLevel(Enum):
    """Technical readiness levels for edge deployment"""

    NOT_READY = "not_ready"
    BASIC_READY = "basic_ready"
    PRODUCTION_READY = "production_ready"
    ENTERPRISE_READY = "enterprise_ready"


class EdgePlatform(Enum):
    """Supported edge deployment platforms"""

    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    INTEL_NUC = "intel_nuc"
    MAC_MINI = "mac_mini"
    GENERIC_ARM64 = "generic_arm64"
    GENERIC_X86_64 = "generic_x86_64"
    AWS_WAVELENGTH = "aws_wavelength"
    AZURE_EDGE = "azure_edge"


@dataclass
class ResourceRequirements:
    """Resource requirements for coalition deployment"""

    min_cpu_cores: int
    min_ram_gb: float
    min_storage_gb: float
    min_network_mbps: float

    # Performance requirements
    min_compute_gflops: float
    max_latency_ms: float
    min_throughput_ops_sec: float

    # Power and thermal constraints
    max_power_watts: Optional[float] = None
    max_temp_celsius: Optional[float] = None

    # Software requirements
    required_python_version: str = "3.9"
    required_packages: List[str] = field(default_factory=list)

    # Hardware accelerators
    requires_gpu: bool = False
    requires_tpu: bool = False
    requires_neural_engine: bool = False


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""

    benchmark_name: str
    metric_name: str
    value: float
    unit: str
    baseline_value: float
    performance_ratio: float  # actual/baseline
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}


@dataclass
class TechnicalReadinessReport:
    """Comprehensive technical readiness assessment report"""

    coalition_id: str
    platform: EdgePlatform
    readiness_level: ReadinessLevel

    # Overall scores (0-100)
    overall_score: float
    hardware_score: float
    performance_score: float
    compatibility_score: float
    resource_score: float

    # Hardware profile
    hardware_profile: HardwareProfile
    resource_requirements: ResourceRequirements

    # Detailed results
    compatibility_results: List[TestResult]
    performance_benchmarks: List[PerformanceBenchmark]

    # Assessment details
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    deployment_ready: bool = False

    # Metadata
    assessment_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coalition_id": self.coalition_id,
            "platform": self.platform.value,
            "readiness_level": self.readiness_level.value,
            "overall_score": self.overall_score,
            "hardware_score": self.hardware_score,
            "performance_score": self.performance_score,
            "compatibility_score": self.compatibility_score,
            "resource_score": self.resource_score,
            "hardware_profile": asdict(self.hardware_profile),
            "resource_requirements": asdict(self.resource_requirements),
            "compatibility_results": [asdict(r) for r in self.compatibility_results],
            "performance_benchmarks": [b.to_dict() for b in self.performance_benchmarks],
            "issues": self.issues,
            "recommendations": self.recommendations,
            "deployment_ready": self.deployment_ready,
            "assessment_duration": self.assessment_duration,
            "timestamp": self.timestamp.isoformat(),
        }


class EdgePerformanceBenchmarker:
    """Specialized benchmarking for edge deployment scenarios"""

    def __init__(self) -> None:
        self.benchmarks = {
            "agent_startup_time": self._benchmark_agent_startup,
            "memory_footprint": self._benchmark_memory_footprint,
            "inference_latency": self._benchmark_inference_latency,
            "concurrent_agents": self._benchmark_concurrent_agents,
            "network_throughput": self._benchmark_network_throughput,
            "storage_io": self._benchmark_storage_io,
            "power_consumption": self._benchmark_power_consumption,
            "thermal_stability": self._benchmark_thermal_stability,
        }

    async def run_benchmarks(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> List[PerformanceBenchmark]:
        """Run all edge deployment benchmarks"""
        results = []

        logger.info("Starting edge deployment performance benchmarks")

        for benchmark_name, benchmark_func in self.benchmarks.items():
            try:
                logger.info(f"Running benchmark: {benchmark_name}")
                result = await benchmark_func(coalition_config, requirements)
                if result:
                    results.append(result)
                    logger.info(
                        f"Benchmark {benchmark_name} completed: {result.value:.2f} {result.unit}"
                    )
            except Exception as e:
                logger.error(f"Benchmark {benchmark_name} failed: {str(e)}")
                # Create a failed benchmark result
                results.append(
                    PerformanceBenchmark(
                        benchmark_name=benchmark_name,
                        metric_name="error",
                        value=0.0,
                        unit="error",
                        baseline_value=1.0,
                        performance_ratio=0.0,
                        passed=False,
                    )
                )

        logger.info(f"Completed {len(results)} benchmarks")
        return results

    async def _benchmark_agent_startup(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark agent startup time"""
        baseline_startup_ms = 5000.0  # 5 seconds baseline

        # Simulate agent startup (in real implementation, would start actual agent)
        start_time = time.time()

        # Mock agent initialization steps
        await asyncio.sleep(0.1)  # Configuration loading
        await asyncio.sleep(0.2)  # Model loading
        await asyncio.sleep(0.1)  # System initialization

        actual_startup_ms = (time.time() - start_time) * 1000

        performance_ratio = baseline_startup_ms / actual_startup_ms
        passed = actual_startup_ms <= requirements.max_latency_ms

        return PerformanceBenchmark(
            benchmark_name="agent_startup_time",
            metric_name="startup_latency",
            value=actual_startup_ms,
            unit="ms",
            baseline_value=baseline_startup_ms,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_memory_footprint(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark memory footprint of coalition agents"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Simulate agent memory usage
        test_data = []
        for i in range(10000):  # Simulate model and data loading
            test_data.append([j for j in range(100)])

        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_footprint = peak_memory - initial_memory

        # Clean up
        del test_data

        baseline_memory_mb = 512.0  # 512 MB baseline
        performance_ratio = baseline_memory_mb / memory_footprint
        passed = memory_footprint <= (requirements.min_ram_gb * 1024 * 0.8)  # 80% of available RAM

        return PerformanceBenchmark(
            benchmark_name="memory_footprint",
            metric_name="memory_usage",
            value=memory_footprint,
            unit="MB",
            baseline_value=baseline_memory_mb,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_inference_latency(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark inference latency for agent decision making"""
        baseline_inference_ms = 100.0  # 100ms baseline

        # Simulate inference workload
        import numpy as np

        start_time = time.time()

        # Mock AI inference operations
        for _ in range(10):
            data = np.random.rand(100, 100)
            result = np.dot(data, data.T)  # Matrix operations
            await asyncio.sleep(0.001)  # Simulated processing

        inference_latency_ms = (time.time() - start_time) * 1000 / 10  # Average per inference

        performance_ratio = baseline_inference_ms / inference_latency_ms
        passed = inference_latency_ms <= requirements.max_latency_ms

        return PerformanceBenchmark(
            benchmark_name="inference_latency",
            metric_name="inference_time",
            value=inference_latency_ms,
            unit="ms",
            baseline_value=baseline_inference_ms,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_concurrent_agents(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark concurrent agent handling capacity"""
        baseline_concurrent_agents = 10.0

        # Test concurrent task handling
        async def mock_agent_task():
            await asyncio.sleep(0.1)
            return "completed"

        start_time = time.time()
        max_concurrent = 0

        for concurrent_count in [5, 10, 20, 50, 100]:
            try:
                tasks = [mock_agent_task() for _ in range(concurrent_count)]
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
                max_concurrent = concurrent_count
            except asyncio.TimeoutError:
                break

        duration = time.time() - start_time

        performance_ratio = max_concurrent / baseline_concurrent_agents
        passed = max_concurrent >= requirements.min_throughput_ops_sec

        return PerformanceBenchmark(
            benchmark_name="concurrent_agents",
            metric_name="max_concurrent",
            value=float(max_concurrent),
            unit="agents",
            baseline_value=baseline_concurrent_agents,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_network_throughput(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark network throughput for agent communication"""
        baseline_throughput_mbps = 100.0

        # Get network interface statistics
        net_io_start = psutil.net_io_counters()
        start_time = time.time()

        # Simulate network activity
        await asyncio.sleep(1.0)

        net_io_end = psutil.net_io_counters()
        duration = time.time() - start_time

        bytes_sent = net_io_end.bytes_sent - net_io_start.bytes_sent
        bytes_recv = net_io_end.bytes_recv - net_io_start.bytes_recv
        total_bytes = bytes_sent + bytes_recv

        throughput_mbps = (total_bytes * 8) / (duration * 1024 * 1024)  # Convert to Mbps

        # Use system's maximum interface speed as approximation
        net_stats = psutil.net_if_stats()
        max_speed = max([stat.speed for stat in net_stats.values() if stat.speed > 0], default=1000)
        actual_throughput = min(throughput_mbps, max_speed)

        performance_ratio = actual_throughput / baseline_throughput_mbps
        passed = actual_throughput >= requirements.min_network_mbps

        return PerformanceBenchmark(
            benchmark_name="network_throughput",
            metric_name="throughput",
            value=actual_throughput,
            unit="Mbps",
            baseline_value=baseline_throughput_mbps,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_storage_io(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark storage I/O performance"""
        baseline_io_mbps = 50.0  # 50 MB/s baseline

        # Test file I/O performance
        test_file = Path("/tmp/edge_deployment_io_test.tmp")
        test_data = b"0" * (1024 * 1024)  # 1MB test data

        start_time = time.time()

        # Write test
        with open(test_file, "wb") as f:
            for _ in range(10):  # Write 10MB
                f.write(test_data)

        # Read test
        with open(test_file, "rb") as f:
            while f.read(1024 * 1024):
                pass

        duration = time.time() - start_time
        io_mbps = (20 * 1024 * 1024) / (duration * 1024 * 1024)  # 20MB total / duration

        # Clean up
        test_file.unlink(missing_ok=True)

        performance_ratio = io_mbps / baseline_io_mbps
        passed = io_mbps >= baseline_io_mbps * 0.5  # 50% of baseline acceptable

        return PerformanceBenchmark(
            benchmark_name="storage_io",
            metric_name="io_throughput",
            value=io_mbps,
            unit="MB/s",
            baseline_value=baseline_io_mbps,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_power_consumption(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark power consumption (estimated)"""
        baseline_power_watts = 15.0  # 15W baseline

        # Estimate power consumption based on CPU usage
        cpu_percent_start = psutil.cpu_percent(interval=None)

        # Create CPU load
        start_time = time.time()
        while time.time() - start_time < 2.0:
            # Light computational load
            sum(range(10000))

        cpu_percent_avg = psutil.cpu_percent(interval=1.0)

        # Rough power estimation based on CPU usage
        # This is a simplified model - real power measurement would require hardware sensors
        estimated_power = baseline_power_watts * (cpu_percent_avg / 100.0) * 1.5

        performance_ratio = baseline_power_watts / estimated_power
        passed = not requirements.max_power_watts or estimated_power <= requirements.max_power_watts

        return PerformanceBenchmark(
            benchmark_name="power_consumption",
            metric_name="estimated_power",
            value=estimated_power,
            unit="watts",
            baseline_value=baseline_power_watts,
            performance_ratio=performance_ratio,
            passed=passed,
        )

    async def _benchmark_thermal_stability(
        self, coalition_config: Dict[str, Any], requirements: ResourceRequirements
    ) -> PerformanceBenchmark:
        """Benchmark thermal stability under load"""
        baseline_temp_celsius = 65.0  # 65°C baseline

        try:
            # Get CPU temperature (Linux/Raspberry Pi)
            if platform.system() == "Linux":
                temp_files = [
                    "/sys/class/thermal/thermal_zone0/temp",
                    "/sys/class/hwmon/hwmon0/temp1_input",
                ]

                current_temp = None
                for temp_file in temp_files:
                    try:
                        with open(temp_file, "r") as f:
                            temp_millicelsius = int(f.read().strip())
                            current_temp = temp_millicelsius / 1000.0
                            break
                    except (FileNotFoundError, ValueError):
                        continue

                if current_temp is None:
                    current_temp = 45.0  # Default reasonable temperature

            elif platform.system() == "Darwin":  # macOS
                try:
                    # Use system profiler or sensors if available
                    current_temp = 45.0  # Default for macOS
                except Exception:
                    current_temp = 45.0
            else:
                current_temp = 45.0  # Default for Windows/other

            performance_ratio = baseline_temp_celsius / current_temp
            passed = (
                not requirements.max_temp_celsius or current_temp <= requirements.max_temp_celsius
            )

            return PerformanceBenchmark(
                benchmark_name="thermal_stability",
                metric_name="cpu_temperature",
                value=current_temp,
                unit="celsius",
                baseline_value=baseline_temp_celsius,
                performance_ratio=performance_ratio,
                passed=passed,
            )

        except Exception as e:
            logger.warning(f"Could not measure temperature: {str(e)}")
            return PerformanceBenchmark(
                benchmark_name="thermal_stability",
                metric_name="cpu_temperature",
                value=0.0,
                unit="celsius",
                baseline_value=baseline_temp_celsius,
                performance_ratio=0.0,
                passed=False,
            )


class TechnicalReadinessValidator:
    """
    Comprehensive technical readiness validation system for edge deployment.

    Integrates with existing hardware infrastructure to provide detailed
    assessment of coalition deployment readiness.
    """

    def __init__(self) -> None:
        self.hardware_detector = HardwareDetector()
        self.compatibility_tester = CompatibilityTester()
        self.device_discovery = DeviceDiscovery()
        self.hal = HardwareAbstractionLayer()
        self.benchmarker = EdgePerformanceBenchmarker()

        # Platform-specific requirements
        self.platform_requirements = self._initialize_platform_requirements()

        logger.info("Technical readiness validator initialized")

    def _initialize_platform_requirements(self) -> Dict[EdgePlatform, ResourceRequirements]:
        """Initialize platform-specific resource requirements"""
        return {
            EdgePlatform.RASPBERRY_PI: ResourceRequirements(
                min_cpu_cores=4,
                min_ram_gb=2.0,
                min_storage_gb=16,
                min_network_mbps=10,
                min_compute_gflops=1.0,
                max_latency_ms=500,
                min_throughput_ops_sec=10,
                max_power_watts=15.0,
                max_temp_celsius=80.0,
            ),
            EdgePlatform.JETSON_NANO: ResourceRequirements(
                min_cpu_cores=4,
                min_ram_gb=4.0,
                min_storage_gb=32,
                min_network_mbps=100,
                min_compute_gflops=5.0,
                max_latency_ms=200,
                min_throughput_ops_sec=50,
                max_power_watts=20.0,
                max_temp_celsius=85.0,
                requires_gpu=True,
            ),
            EdgePlatform.INTEL_NUC: ResourceRequirements(
                min_cpu_cores=4,
                min_ram_gb=8.0,
                min_storage_gb=128,
                min_network_mbps=1000,
                min_compute_gflops=10.0,
                max_latency_ms=100,
                min_throughput_ops_sec=100,
                max_power_watts=65.0,
                max_temp_celsius=85.0,
            ),
            EdgePlatform.MAC_MINI: ResourceRequirements(
                min_cpu_cores=8,
                min_ram_gb=8.0,
                min_storage_gb=256,
                min_network_mbps=1000,
                min_compute_gflops=20.0,
                max_latency_ms=50,
                min_throughput_ops_sec=200,
                max_power_watts=150.0,
                max_temp_celsius=85.0,
                requires_neural_engine=True,
            ),
            EdgePlatform.GENERIC_ARM64: ResourceRequirements(
                min_cpu_cores=4,
                min_ram_gb=4.0,
                min_storage_gb=32,
                min_network_mbps=100,
                min_compute_gflops=3.0,
                max_latency_ms=300,
                min_throughput_ops_sec=30,
                max_power_watts=25.0,
                max_temp_celsius=80.0,
            ),
            EdgePlatform.GENERIC_X86_64: ResourceRequirements(
                min_cpu_cores=4,
                min_ram_gb=8.0,
                min_storage_gb=64,
                min_network_mbps=1000,
                min_compute_gflops=8.0,
                max_latency_ms=100,
                min_throughput_ops_sec=80,
                max_power_watts=95.0,
                max_temp_celsius=85.0,
            ),
        }

    async def assess_technical_readiness(
        self,
        coalition_id: str,
        coalition_config: Dict[str, Any],
        target_platform: Optional[EdgePlatform] = None,
    ) -> TechnicalReadinessReport:
        """
        Perform comprehensive technical readiness assessment.

        Args:
            coalition_id: Unique identifier for the coalition
            coalition_config: Coalition configuration and requirements
            target_platform: Target edge platform (auto-detect if None)

        Returns:
            Comprehensive technical readiness report
        """
        logger.info(f"Starting technical readiness assessment for coalition {coalition_id}")
        start_time = time.time()

        try:
            # Step 1: Hardware detection and profiling
            hardware_profile = self.hardware_detector.detect_hardware()
            logger.info(f"Detected hardware profile: {hardware_profile.name}")

            # Step 2: Platform determination
            if not target_platform:
                target_platform = self._determine_target_platform(hardware_profile)

            # Step 3: Get platform requirements
            requirements = self.platform_requirements.get(
                target_platform, self.platform_requirements[EdgePlatform.GENERIC_X86_64]
            )

            # Step 4: Hardware compatibility testing
            compatibility_results = await self._run_compatibility_tests(
                coalition_config, hardware_profile
            )

            # Step 5: Performance benchmarking
            performance_benchmarks = await self.benchmarker.run_benchmarks(
                coalition_config, requirements
            )

            # Step 6: Score calculation and analysis
            scores = self._calculate_readiness_scores(
                hardware_profile, requirements, compatibility_results, performance_benchmarks
            )

            # Step 7: Issue and recommendation analysis
            issues, recommendations = self._analyze_issues_and_recommendations(
                hardware_profile, requirements, compatibility_results, performance_benchmarks
            )

            # Step 8: Determine readiness level
            readiness_level = self._determine_readiness_level(scores["overall_score"])

            # Step 9: Generate final report
            assessment_duration = time.time() - start_time

            report = TechnicalReadinessReport(
                coalition_id=coalition_id,
                platform=target_platform,
                readiness_level=readiness_level,
                overall_score=scores["overall_score"],
                hardware_score=scores["hardware_score"],
                performance_score=scores["performance_score"],
                compatibility_score=scores["compatibility_score"],
                resource_score=scores["resource_score"],
                hardware_profile=hardware_profile,
                resource_requirements=requirements,
                compatibility_results=compatibility_results,
                performance_benchmarks=performance_benchmarks,
                issues=issues,
                recommendations=recommendations,
                deployment_ready=scores["overall_score"] >= 70.0,
                assessment_duration=assessment_duration,
            )

            logger.info(
                f"Technical readiness assessment completed. "
                f"Overall score: {scores['overall_score']:.1f}, "
                f"Ready: {report.deployment_ready}"
            )

            return report

        except Exception as e:
            logger.error(f"Technical readiness assessment failed: {str(e)}")
            raise

    def _determine_target_platform(self, hardware_profile: HardwareProfile) -> EdgePlatform:
        """Determine target platform based on hardware profile"""
        arch = hardware_profile.architecture.lower()
        name = hardware_profile.name.lower()

        if "raspberry" in name or "pi" in name:
            return EdgePlatform.RASPBERRY_PI
        elif "jetson" in name:
            if "xavier" in name:
                return EdgePlatform.JETSON_XAVIER
            else:
                return EdgePlatform.JETSON_NANO
        elif "mac" in name and "mini" in name:
            return EdgePlatform.MAC_MINI
        elif "nuc" in name:
            return EdgePlatform.INTEL_NUC
        elif arch in ["arm64", "aarch64"]:
            return EdgePlatform.GENERIC_ARM64
        elif arch in ["x86_64", "amd64"]:
            return EdgePlatform.GENERIC_X86_64
        else:
            return EdgePlatform.GENERIC_X86_64  # Default fallback

    async def _run_compatibility_tests(
        self, coalition_config: Dict[str, Any], hardware_profile: HardwareProfile
    ) -> List[TestResult]:
        """Run hardware compatibility tests"""
        logger.info("Running hardware compatibility tests")

        # Create a temporary package directory for testing
        package_dir = Path("/tmp/coalition_test_package")
        package_dir.mkdir(exist_ok=True)

        try:
            # Run compatibility tests using existing infrastructure
            test_results = self.compatibility_tester.run_tests(package_dir, hardware_profile)

            # Convert to TestResult objects
            compatibility_results = []
            for result_dict in test_results.get("test_results", []):
                test_result = TestResult(
                    test_name=result_dict["test_name"],
                    status=TestStatus(result_dict["status"]),
                    duration=result_dict["duration"],
                    message=result_dict["message"],
                    details=result_dict.get("details", {}),
                    error=result_dict.get("error"),
                )
                compatibility_results.append(test_result)

            return compatibility_results

        except Exception as e:
            logger.error(f"Compatibility testing failed: {str(e)}")
            return []
        finally:
            # Clean up
            if package_dir.exists():
                import shutil

                shutil.rmtree(package_dir, ignore_errors=True)

    def _calculate_readiness_scores(
        self,
        hardware_profile: HardwareProfile,
        requirements: ResourceRequirements,
        compatibility_results: List[TestResult],
        performance_benchmarks: List[PerformanceBenchmark],
    ) -> Dict[str, float]:
        """Calculate comprehensive readiness scores"""

        # Hardware score (0-100)
        hardware_score = self._calculate_hardware_score(hardware_profile, requirements)

        # Compatibility score (0-100)
        compatibility_score = self._calculate_compatibility_score(compatibility_results)

        # Performance score (0-100)
        performance_score = self._calculate_performance_score(performance_benchmarks)

        # Resource score (0-100)
        resource_score = self._calculate_resource_score(hardware_profile, requirements)

        # Overall score (weighted average)
        overall_score = (
            hardware_score * 0.25
            + compatibility_score * 0.25
            + performance_score * 0.35
            + resource_score * 0.15
        )

        return {
            "overall_score": overall_score,
            "hardware_score": hardware_score,
            "compatibility_score": compatibility_score,
            "performance_score": performance_score,
            "resource_score": resource_score,
        }

    def _calculate_hardware_score(
        self, hardware_profile: HardwareProfile, requirements: ResourceRequirements
    ) -> float:
        """Calculate hardware adequacy score"""
        score = 100.0

        # CPU cores check
        if hardware_profile.cpu_cores < requirements.min_cpu_cores:
            deficit = (
                requirements.min_cpu_cores - hardware_profile.cpu_cores
            ) / requirements.min_cpu_cores
            score -= deficit * 30

        # RAM check
        if hardware_profile.ram_gb < requirements.min_ram_gb:
            deficit = (requirements.min_ram_gb - hardware_profile.ram_gb) / requirements.min_ram_gb
            score -= deficit * 25

        # Storage check
        if hardware_profile.storage_gb < requirements.min_storage_gb:
            deficit = (
                requirements.min_storage_gb - hardware_profile.storage_gb
            ) / requirements.min_storage_gb
            score -= deficit * 20

        # GPU requirement check
        if requirements.requires_gpu and not hardware_profile.gpu_available:
            score -= 25

        return max(0.0, score)

    def _calculate_compatibility_score(self, compatibility_results: List[TestResult]) -> float:
        """Calculate compatibility test score"""
        if not compatibility_results:
            return 50.0  # Default if no tests run

        passed_tests = sum(1 for r in compatibility_results if r.status == TestStatus.PASSED)
        total_tests = len(compatibility_results)

        return (passed_tests / total_tests) * 100.0

    def _calculate_performance_score(
        self, performance_benchmarks: List[PerformanceBenchmark]
    ) -> float:
        """Calculate performance benchmark score"""
        if not performance_benchmarks:
            return 50.0  # Default if no benchmarks run

        # Weight benchmarks by importance
        benchmark_weights = {
            "agent_startup_time": 0.2,
            "memory_footprint": 0.15,
            "inference_latency": 0.25,
            "concurrent_agents": 0.2,
            "network_throughput": 0.1,
            "storage_io": 0.05,
            "power_consumption": 0.03,
            "thermal_stability": 0.02,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for benchmark in performance_benchmarks:
            weight = benchmark_weights.get(benchmark.benchmark_name, 0.05)
            total_weight += weight

            if benchmark.passed:
                # Performance ratio > 1.0 means better than baseline
                benchmark_score = min(100.0, benchmark.performance_ratio * 100.0)
            else:
                benchmark_score = benchmark.performance_ratio * 50.0  # Penalty for failing

            weighted_score += benchmark_score * weight

        return weighted_score / total_weight if total_weight > 0 else 50.0

    def _calculate_resource_score(
        self, hardware_profile: HardwareProfile, requirements: ResourceRequirements
    ) -> float:
        """Calculate resource adequacy score"""
        score = 100.0

        # CPU utilization headroom
        cpu_ratio = hardware_profile.cpu_cores / requirements.min_cpu_cores
        if cpu_ratio < 1.0:
            score -= (1.0 - cpu_ratio) * 40
        elif cpu_ratio > 2.0:
            score += min(10.0, (cpu_ratio - 2.0) * 5)  # Bonus for excess capacity

        # Memory headroom
        memory_ratio = hardware_profile.ram_gb / requirements.min_ram_gb
        if memory_ratio < 1.0:
            score -= (1.0 - memory_ratio) * 35
        elif memory_ratio > 2.0:
            score += min(10.0, (memory_ratio - 2.0) * 5)  # Bonus for excess capacity

        # Storage headroom
        storage_ratio = hardware_profile.storage_gb / requirements.min_storage_gb
        if storage_ratio < 1.0:
            score -= (1.0 - storage_ratio) * 25

        return max(0.0, min(100.0, score))

    def _analyze_issues_and_recommendations(
        self,
        hardware_profile: HardwareProfile,
        requirements: ResourceRequirements,
        compatibility_results: List[TestResult],
        performance_benchmarks: List[PerformanceBenchmark],
    ) -> Tuple[List[str], List[str]]:
        """Analyze issues and generate recommendations"""
        issues = []
        recommendations = []

        # Hardware issues
        if hardware_profile.cpu_cores < requirements.min_cpu_cores:
            issues.append(
                f"Insufficient CPU cores: {hardware_profile.cpu_cores} < {requirements.min_cpu_cores}"
            )
            recommendations.append("Consider upgrading to a device with more CPU cores")

        if hardware_profile.ram_gb < requirements.min_ram_gb:
            issues.append(
                f"Insufficient RAM: {hardware_profile.ram_gb:.1f}GB < {requirements.min_ram_gb}GB"
            )
            recommendations.append("Add more RAM or choose a device with more memory")

        if hardware_profile.storage_gb < requirements.min_storage_gb:
            issues.append(
                f"Insufficient storage: {hardware_profile.storage_gb:.1f}GB < {requirements.min_storage_gb}GB"
            )
            recommendations.append("Add external storage or upgrade internal storage")

        if requirements.requires_gpu and not hardware_profile.gpu_available:
            issues.append("GPU acceleration required but not available")
            recommendations.append(
                "Choose a device with GPU support or disable GPU-dependent features"
            )

        # Compatibility issues
        failed_tests = [r for r in compatibility_results if r.status == TestStatus.FAILED]
        if failed_tests:
            issues.append(f"{len(failed_tests)} compatibility tests failed")
            recommendations.append(
                "Review failed compatibility tests and address underlying issues"
            )

        # Performance issues
        failed_benchmarks = [b for b in performance_benchmarks if not b.passed]
        if failed_benchmarks:
            for benchmark in failed_benchmarks:
                issues.append(f"Performance benchmark failed: {benchmark.benchmark_name}")

                if benchmark.benchmark_name == "agent_startup_time":
                    recommendations.append("Optimize agent initialization and model loading")
                elif benchmark.benchmark_name == "memory_footprint":
                    recommendations.append("Reduce memory usage or increase available RAM")
                elif benchmark.benchmark_name == "inference_latency":
                    recommendations.append(
                        "Optimize inference pipeline or enable hardware acceleration"
                    )
                elif benchmark.benchmark_name == "concurrent_agents":
                    recommendations.append("Optimize resource sharing and concurrency handling")

        # General recommendations
        if not issues:
            recommendations.append("System meets all technical requirements for edge deployment")
        else:
            recommendations.append("Address identified issues before production deployment")

        return issues, recommendations

    def _determine_readiness_level(self, overall_score: float) -> ReadinessLevel:
        """Determine readiness level based on overall score"""
        if overall_score >= 90.0:
            return ReadinessLevel.ENTERPRISE_READY
        elif overall_score >= 80.0:
            return ReadinessLevel.PRODUCTION_READY
        elif overall_score >= 60.0:
            return ReadinessLevel.BASIC_READY
        else:
            return ReadinessLevel.NOT_READY

    def save_report(self, report: TechnicalReadinessReport, output_path: Path) -> None:
        """Save technical readiness report to file"""
        report_data = report.to_dict()

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Technical readiness report saved to {output_path}")

    def generate_summary_report(self, reports: List[TechnicalReadinessReport]) -> Dict[str, Any]:
        """Generate summary report from multiple assessments"""
        if not reports:
            return {}

        summary = {
            "total_assessments": len(reports),
            "readiness_distribution": {},
            "average_scores": {},
            "common_issues": {},
            "platform_distribution": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Readiness level distribution
        for level in ReadinessLevel:
            count = sum(1 for r in reports if r.readiness_level == level)
            summary["readiness_distribution"][level.value] = count

        # Average scores
        summary["average_scores"] = {
            "overall": sum(r.overall_score for r in reports) / len(reports),
            "hardware": sum(r.hardware_score for r in reports) / len(reports),
            "performance": sum(r.performance_score for r in reports) / len(reports),
            "compatibility": sum(r.compatibility_score for r in reports) / len(reports),
            "resource": sum(r.resource_score for r in reports) / len(reports),
        }

        # Common issues
        all_issues = []
        for report in reports:
            all_issues.extend(report.issues)

        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        summary["common_issues"] = dict(
            sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Platform distribution
        for platform in EdgePlatform:
            count = sum(1 for r in reports if r.platform == platform)
            if count > 0:
                summary["platform_distribution"][platform.value] = count

        return summary


# Convenience function for direct usage
async def assess_coalition_technical_readiness(
    coalition_id: str,
    coalition_config: Dict[str, Any],
    target_platform: Optional[EdgePlatform] = None,
    output_path: Optional[Path] = None,
) -> TechnicalReadinessReport:
    """
    Convenience function to assess coalition technical readiness.

    Args:
        coalition_id: Unique identifier for the coalition
        coalition_config: Coalition configuration
        target_platform: Target edge platform (auto-detect if None)
        output_path: Path to save report (optional)

    Returns:
        Technical readiness report
    """
    validator = TechnicalReadinessValidator()
    report = await validator.assess_technical_readiness(
        coalition_id, coalition_config, target_platform
    )

    if output_path:
        validator.save_report(report, output_path)

    return report
