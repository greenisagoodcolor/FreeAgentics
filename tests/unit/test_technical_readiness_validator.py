"""
Comprehensive tests for Technical Readiness Validation for Edge Deployment.

Tests the sophisticated technical readiness assessment system that validates coalition
technical readiness for edge deployment, integrating with hardware infrastructure
and providing detailed performance benchmarks.
"""

import asyncio

# Mock infrastructure modules before importing the main module
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Mock infrastructure modules that may not be available
sys.modules["infrastructure.hardware.device_discovery"] = Mock()
sys.modules["infrastructure.hardware.hal_core"] = Mock()
sys.modules["infrastructure.hardware.offline_capabilities"] = Mock()
sys.modules["infrastructure.hardware.resource_manager"] = Mock()
sys.modules["infrastructure.deployment.hardware_compatibility"] = Mock()


# Create mock classes for the imports
class MockHardwareProfile:
    def __init__(self, name, architecture, cpu_count, memory_gb, disk_gb, accelerators):
        self.name = name
        self.architecture = architecture
        self.cpu_count = cpu_count
        self.cpu_cores = cpu_count  # Add this attribute that tests expect
        self.memory_gb = memory_gb
        self.ram_gb = memory_gb  # Expected by validator
        self.disk_gb = disk_gb
        self.storage_gb = disk_gb  # Expected by validator
        self.accelerators = accelerators
        self.gpu_available = len(accelerators) > 0  # Expected by validator


class MockTestStatus:
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class MockTestResult:
    def __init__(self, test_name, status, duration, message, details=None, error=None):
        self.test_name = test_name
        self.status = status
        self.duration = duration
        self.message = message
        self.details = details or {}
        self.error = error


# Mock the hardware compatibility module
class MockCompatibilityTester:
    async def run_tests(self, *args, **kwargs):
        return [
            MockTestResult("python_version", MockTestStatus.PASSED, 0.1, "Python version OK"),
            MockTestResult("memory_test", MockTestStatus.PASSED, 0.2, "Memory test OK"),
        ]


class MockHardwareDetector:
    def detect_hardware(self):
        return MockHardwareProfile("TestDevice", "x86_64", 8, 16.0, 512.0, [])


# Set up mock modules with required classes
sys.modules["infrastructure.deployment.hardware_compatibility"].HardwareProfile = (
    MockHardwareProfile
)
sys.modules["infrastructure.deployment.hardware_compatibility"].TestResult = MockTestResult
sys.modules["infrastructure.deployment.hardware_compatibility"].TestStatus = MockTestStatus
sys.modules["infrastructure.deployment.hardware_compatibility"].CompatibilityTester = (
    MockCompatibilityTester
)
sys.modules["infrastructure.deployment.hardware_compatibility"].HardwareDetector = (
    MockHardwareDetector
)
sys.modules["infrastructure.hardware.device_discovery"].DeviceDiscovery = Mock
sys.modules["infrastructure.hardware.device_discovery"].DeviceType = Mock
sys.modules["infrastructure.hardware.hal_core"].HardwareAbstractionLayer = Mock
sys.modules["infrastructure.hardware.hal_core"].HardwareType = Mock
sys.modules["infrastructure.hardware.hal_core"].ResourceConstraints = Mock

from coalitions.readiness.technical_readiness_validator import (
    EdgePerformanceBenchmarker,
    EdgePlatform,
    PerformanceBenchmark,
    ReadinessLevel,
    ResourceRequirements,
    TechnicalReadinessReport,
    TechnicalReadinessValidator,
)

# Use mock classes in tests
HardwareProfile = MockHardwareProfile
TestResult = MockTestResult
TestStatus = MockTestStatus


class TestReadinessLevel:
    """Test ReadinessLevel enum functionality."""

    def test_all_readiness_levels_defined(self):
        """Test that all expected readiness levels are defined."""
        expected_levels = {"NOT_READY", "BASIC_READY", "PRODUCTION_READY", "ENTERPRISE_READY"}

        actual_levels = {level.name for level in ReadinessLevel}
        assert actual_levels == expected_levels

    def test_readiness_level_values(self):
        """Test readiness level string values."""
        assert ReadinessLevel.NOT_READY.value == "not_ready"
        assert ReadinessLevel.BASIC_READY.value == "basic_ready"
        assert ReadinessLevel.PRODUCTION_READY.value == "production_ready"
        assert ReadinessLevel.ENTERPRISE_READY.value == "enterprise_ready"


class TestEdgePlatform:
    """Test EdgePlatform enum functionality."""

    def test_all_edge_platforms_defined(self):
        """Test that all expected edge platforms are defined."""
        expected_platforms = {
            "RASPBERRY_PI",
            "JETSON_NANO",
            "JETSON_XAVIER",
            "INTEL_NUC",
            "MAC_MINI",
            "GENERIC_ARM64",
            "GENERIC_X86_64",
            "AWS_WAVELENGTH",
            "AZURE_EDGE",
        }

        actual_platforms = {platform.name for platform in EdgePlatform}
        assert actual_platforms == expected_platforms

    def test_edge_platform_values(self):
        """Test edge platform string values."""
        assert EdgePlatform.RASPBERRY_PI.value == "raspberry_pi"
        assert EdgePlatform.JETSON_NANO.value == "jetson_nano"
        assert EdgePlatform.INTEL_NUC.value == "intel_nuc"
        assert EdgePlatform.MAC_MINI.value == "mac_mini"


class TestEdgePerformanceBenchmarker:
    """Test EdgePerformanceBenchmarker functionality."""

    def test_benchmarker_initialization(self):
        """Test benchmarker initialization."""
        benchmarker = EdgePerformanceBenchmarker()

        expected_benchmarks = {
            "agent_startup_time",
            "memory_footprint",
            "inference_latency",
            "concurrent_agents",
            "network_throughput",
            "storage_io",
            "power_consumption",
            "thermal_stability",
        }

        assert set(benchmarker.benchmarks.keys()) == expected_benchmarks

    @pytest.mark.asyncio
    async def test_benchmark_agent_startup_time(self):
        """Test agent startup time benchmark."""
        benchmarker = EdgePerformanceBenchmarker()

        mock_config = {"agents": [{"name": "test_agent"}]}
        requirements = ResourceRequirements(
            min_cpu_cores=4,
            min_ram_gb=2.0,
            min_storage_gb=16,
            min_network_mbps=10,
            min_compute_gflops=1.0,
            max_latency_ms=5000,
            min_throughput_ops_sec=10,
        )

        result = await benchmarker._benchmark_agent_startup(mock_config, requirements)

        assert result.benchmark_name == "agent_startup_time"
        assert result.metric_name == "startup_latency"
        assert result.unit == "ms"
        assert result.value > 0
        assert result.baseline_value == 5000.0
        assert result.performance_ratio > 0
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_benchmark_memory_footprint(self):
        """Test memory footprint benchmark."""
        benchmarker = EdgePerformanceBenchmarker()

        mock_config = {"agents": [{"name": "test_agent"}]}
        requirements = ResourceRequirements(
            min_cpu_cores=4,
            min_ram_gb=8.0,
            min_storage_gb=16,
            min_network_mbps=10,
            min_compute_gflops=1.0,
            max_latency_ms=5000,
            min_throughput_ops_sec=10,
        )

        result = await benchmarker._benchmark_memory_footprint(mock_config, requirements)

        assert result.benchmark_name == "memory_footprint"
        assert result.metric_name == "memory_usage"
        assert result.unit == "MB"
        assert result.value >= 0
        assert result.baseline_value == 512.0
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_benchmark_inference_latency(self):
        """Test inference latency benchmark."""
        benchmarker = EdgePerformanceBenchmarker()

        mock_config = {"agents": [{"name": "test_agent"}]}
        requirements = ResourceRequirements(
            min_cpu_cores=4,
            min_ram_gb=2.0,
            min_storage_gb=16,
            min_network_mbps=10,
            min_compute_gflops=1.0,
            max_latency_ms=5000,
            min_throughput_ops_sec=10,
        )

        result = await benchmarker._benchmark_inference_latency(mock_config, requirements)

        assert result.benchmark_name == "inference_latency"
        assert result.metric_name == "inference_time"
        assert result.unit == "ms"
        assert result.value > 0
        assert result.baseline_value == 100.0
        assert result.performance_ratio > 0
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_run_benchmarks_basic(self):
        """Test basic benchmark execution."""
        benchmarker = EdgePerformanceBenchmarker()

        mock_config = {"agents": [{"name": "test_agent"}]}
        requirements = ResourceRequirements(
            min_cpu_cores=4,
            min_ram_gb=2.0,
            min_storage_gb=16,
            min_network_mbps=10,
            min_compute_gflops=1.0,
            max_latency_ms=5000,
            min_throughput_ops_sec=10,
        )

        results = await benchmarker.run_benchmarks(mock_config, requirements)

        # Should have 8 benchmarks
        assert len(results) == 8

        # Check benchmark names
        benchmark_names = {r.benchmark_name for r in results}
        expected_names = {
            "agent_startup_time",
            "memory_footprint",
            "inference_latency",
            "concurrent_agents",
            "network_throughput",
            "storage_io",
            "power_consumption",
            "thermal_stability",
        }
        assert benchmark_names == expected_names


class TestTechnicalReadinessValidator:
    """Test TechnicalReadinessValidator functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = TechnicalReadinessValidator()

        # Check that platform requirements are initialized
        assert len(validator.platform_requirements) > 0

        # Check specific platform requirements
        assert EdgePlatform.RASPBERRY_PI in validator.platform_requirements
        assert EdgePlatform.INTEL_NUC in validator.platform_requirements

        # Check that components are initialized
        assert hasattr(validator, "hardware_detector")
        assert hasattr(validator, "compatibility_tester")
        assert hasattr(validator, "device_discovery")
        assert hasattr(validator, "hal")
        assert hasattr(validator, "benchmarker")

    def test_platform_requirements_raspberry_pi(self):
        """Test Raspberry Pi platform requirements."""
        validator = TechnicalReadinessValidator()
        requirements = validator.platform_requirements[EdgePlatform.RASPBERRY_PI]

        assert requirements.min_cpu_cores == 4
        assert requirements.min_ram_gb == 2.0
        assert requirements.min_storage_gb == 16
        assert requirements.max_power_watts == 15.0
        assert requirements.max_temp_celsius == 80.0

    def test_platform_requirements_intel_nuc(self):
        """Test Intel NUC platform requirements."""
        validator = TechnicalReadinessValidator()
        requirements = validator.platform_requirements[EdgePlatform.INTEL_NUC]

        assert requirements.min_cpu_cores == 4
        assert requirements.min_ram_gb == 8.0
        assert requirements.min_storage_gb == 128
        assert requirements.max_power_watts == 65.0
        assert requirements.max_temp_celsius == 85.0

    def test_determine_target_platform_intel_nuc(self):
        """Test platform determination for Intel NUC."""
        validator = TechnicalReadinessValidator()

        # Create hardware profile that matches Intel NUC
        hardware_profile = HardwareProfile("Intel NUC", "x86_64", 8, 16.0, 512.0, [])

        platform = validator._determine_target_platform(hardware_profile)

        # Should detect as Intel NUC or generic x86_64
        assert platform in [EdgePlatform.INTEL_NUC, EdgePlatform.GENERIC_X86_64]

    @pytest.mark.asyncio
    async def test_assess_technical_readiness_basic(self):
        """Test basic technical readiness assessment."""
        validator = TechnicalReadinessValidator()

        # Mock the hardware detector to return a known profile
        with patch.object(validator.hardware_detector, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareProfile("TestDevice", "x86_64", 8, 16.0, 256.0, [])

            # Mock compatibility tester
            with patch.object(validator.compatibility_tester, "run_tests") as mock_compat:
                mock_compat.return_value = [
                    TestResult("test1", TestStatus.PASSED, 0.1, "OK"),
                    TestResult("test2", TestStatus.PASSED, 0.1, "OK"),
                ]

                coalition_config = {"agents": [{"name": "test_agent"}]}

                report = await validator.assess_technical_readiness(
                    "test_coalition", coalition_config
                )

                assert report.coalition_id == "test_coalition"
                assert report.platform in EdgePlatform
                assert report.readiness_level in ReadinessLevel
                assert report.overall_score >= 0  # Score can exceed 100 if performance is excellent
                assert isinstance(report.deployment_ready, bool)
                assert report.assessment_duration > 0
